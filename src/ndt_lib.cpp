#include "ndt_matching/ndt_lib.h"

namespace ndt_matching{

NdtLib::NdtLib()
{

	max_iterations_ = 35;
	nr_iterations_ = 0;
	real_iterations_ = 0;	
	
	converged_ = false;
	map_loaded_ = false;

	transformation_epsilon_ = 0.1;
	outlier_ratio_ = 0.55;
	step_size_ = 0.1; 
	resolution_ = 1.0f;

	final_transformation_ = Eigen::Matrix<float, 4, 4>::Identity();
	trans_cloud_.points.clear();
	trans_probability_ = 0;

	// Initializes the gaussian fitting parameters (eq. 6.8) [Magnusson 2009]
	double gauss_c1, gauss_c2, gauss_d3;
	gauss_c1 = 10.0 * (1 - outlier_ratio_);
	gauss_c2 = outlier_ratio_ / pow (resolution_, 3);
	gauss_d3 = -log (gauss_c2);

	gauss_d1_ = -log ( gauss_c1 + gauss_c2 ) - gauss_d3;
	gauss_d2_ = -2 * log ((-log ( gauss_c1 * exp ( -0.5 ) + gauss_c2 ) - gauss_d3) / gauss_d1_);


} /* ctor */

void NdtLib::align(const Eigen::Matrix<float, 4, 4> guess)
{


	converged_ = false;

	final_transformation_ = transformation_ = previous_transformation_ = Eigen::Matrix<float, 4, 4>::Identity();

	trans_cloud_.points.resize(source_cloud_->points.size());

	for (size_t i = 0; i < trans_cloud_.points.size(); i++) {
		trans_cloud_.points[i] = source_cloud_->points[i];
	}



	std::chrono::time_point<std::chrono::system_clock> align_start, align_end;
	align_start = std::chrono::system_clock::now();
	computeTransformation(guess);	
	align_end = std::chrono::system_clock::now();
	double align_time = std::chrono::duration_cast<std::chrono::microseconds>(align_end - align_start).count() / 1000.0;

	std::chrono::time_point<std::chrono::system_clock> fitness_start, fitness_end;
	fitness_start = std::chrono::system_clock::now();
	double fitness_score = getFitnessScore();
	fitness_end = std::chrono::system_clock::now();
	double fitness_time = std::chrono::duration_cast<std::chrono::microseconds>(fitness_end - fitness_start).count() / 1000.0;



    std::cout << "-----------------------------------------------------------------" << std::endl;
    std::cout << "Number of Scan Points: " << source_cloud_->size() << " points." << std::endl;
    std::cout << "NDT has converged: " << converged_ << std::endl;
    std::cout << "Fitness Score: " << fitness_score << std::endl;
    std::cout << "Get fitness score time: " << fitness_time << std::endl;
    std::cout << "Transformation Probability: " << trans_probability_ << std::endl;
    std::cout << "Execution Time: " << align_time << " ms." << std::endl;
    std::cout << "Number of Iterations: " << nr_iterations_ << std::endl;
    std::cout << "Transformation Matrix: " << std::endl;
    std::cout << final_transformation_ << std::endl;
    std::cout << "-----------------------------------------------------------------" << std::endl;

}

void NdtLib::setInputSource(const pcl::PointCloud<PointSourceType>::Ptr& input)
{
	source_cloud_ = std::make_shared< pcl::PointCloud<PointSourceType> >();
	*source_cloud_ = *input;
}

void NdtLib::setInputTarget(pcl::PointCloud<PointTargetType>::Ptr input)
{
	if (target_cloud_ != nullptr){
		std::cout<<"Map is already loaded! Changing existing map."<<std::endl;
	}
	
	std::cout<<"Building voxel grid of the map. Resolution:"<<resolution_<<std::endl;
	std::chrono::time_point<std::chrono::system_clock> voxel_start, voxel_end;

	voxel_start = std::chrono::system_clock::now();

	// Build the voxel grid
	if (input->points.size() > 0) {
		voxel_grid_.setLeafSize(resolution_, resolution_, resolution_);
		voxel_grid_.setInput(input);
	}

	voxel_end = std::chrono::system_clock::now();
	double voxel_time = std::chrono::duration_cast<std::chrono::microseconds>(voxel_end - voxel_start).count() / 1000.0;
	std::cout<<"Building Voxel grid finished: build time:"<<voxel_time<<" milliseconds"<<std::endl;
	std::cout<<"Total Voxel Number:"<<voxel_grid_.getVoxelNum()<<std::endl;

	target_cloud_ = input;
	map_loaded_ = true;
}


bool NdtLib::isMapLoaded() const{
	return map_loaded_;
}

bool NdtLib::isConverged() const{
	return converged_;
}

Eigen::Matrix<float, 4, 4> NdtLib::getFinalTransformation() const
{
	return final_transformation_;
}

double NdtLib::getTransformationProbability() const
{
	return trans_probability_;
}

void NdtLib::setStepSize(double step_size)
{
	step_size_ = step_size;
}

void NdtLib::setResolution(float resolution)
{
	resolution_ = resolution;
}

void NdtLib::setOutlierRatio(double olr)
{
	outlier_ratio_ = olr;
}

double NdtLib::getStepSize() const
{
	return step_size_;
}

float NdtLib::getResolution() const
{
	return resolution_;
}

double NdtLib::getOutlierRatio() const
{
	return outlier_ratio_;
}

void NdtLib::setTransformationEpsilon(double trans_eps)
{
	transformation_epsilon_ = trans_eps;
}

double NdtLib::getTransformationEpsilon() const
{
	return transformation_epsilon_;
}

void NdtLib::setMaximumIterations(int max_itr)
{
	max_iterations_ = max_itr;
}

int NdtLib::getMaximumIterations() const
{
	return max_iterations_;
}

int NdtLib::getFinalNumIteration() const
{
	return nr_iterations_;
}

int NdtLib::getRealIterations() const
{
	return real_iterations_;
}

double NdtLib::getFitnessScore(double max_range)
{
	double fitness_score = 0.0;

	pcl::PointCloud<PointSourceType> trans_cloud;

	pcl::transformPointCloud(*source_cloud_, trans_cloud, final_transformation_);

	double distance;
	int nr = 0;

	for (size_t i = 0; i < trans_cloud.points.size(); i++) {
		PointSourceType q = trans_cloud.points[i];

		distance = voxel_grid_.nearestNeighborDistance(q, max_range);

		if (distance < max_range) {
			fitness_score += distance;
			nr++;
		}
	}

	if (nr > 0) {
		return (fitness_score / nr);
	}

	return DBL_MAX;
}


void NdtLib::computeTransformation(const Eigen::Matrix<float, 4, 4> &guess)
{
	
	nr_iterations_ = 0;
	converged_ = false;

	double gauss_c1, gauss_c2, gauss_d3;

	gauss_c1 = 10 * ( 1 - outlier_ratio_);
	gauss_c2 = outlier_ratio_ / pow(resolution_, 3);
	gauss_d3 = - log(gauss_c2);
	gauss_d1_ = -log(gauss_c1 + gauss_c2) - gauss_d3;
	gauss_d2_ = -2 * log((-log(gauss_c1 * exp(-0.5) + gauss_c2) - gauss_d3) / gauss_d1_);

    // std::cout << "Guess Transformation Matrix: " << std::endl;
    // std::cout << guess << std::endl;

	if (guess != Eigen::Matrix4f::Identity()) {
		final_transformation_ = guess;

		pcl::transformPointCloud(*source_cloud_, trans_cloud_, guess);
	}

	Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor> eig_transformation;
	eig_transformation.matrix() = final_transformation_;

	Eigen::Matrix<double, 6, 1> p, delta_p, score_gradient;
	Eigen::Vector3f init_translation = eig_transformation.translation();
	Eigen::Vector3f init_rotation = eig_transformation.rotation().eulerAngles(0, 1, 2);

	p << init_translation(0), init_translation(1), init_translation(2), init_rotation(0), init_rotation(1), init_rotation(2);

	Eigen::Matrix<double, 6, 6> hessian;

	double score = 0;
	double delta_p_norm;



	std::chrono::time_point<std::chrono::system_clock> derivative_compute_start, derivative_compute_end;
	derivative_compute_start = std::chrono::system_clock::now();
	score = computeDerivatives(score_gradient, hessian, trans_cloud_, p);
	derivative_compute_end = std::chrono::system_clock::now();
	double derivative_compute_time = std::chrono::duration_cast<std::chrono::microseconds>(derivative_compute_end - derivative_compute_start).count() / 1000.0;
	std::cout << "derivative_compute Time: " << derivative_compute_time << " ms." << std::endl;


	int points_number = source_cloud_->points.size();


	while (!converged_) {
		previous_transformation_ = transformation_;

		Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6> > sv(hessian, Eigen::ComputeFullU | Eigen::ComputeFullV);

		delta_p = sv.solve(-score_gradient);

		delta_p_norm = delta_p.norm();

		if (delta_p_norm == 0 || delta_p_norm != delta_p_norm) {
			trans_probability_ = score / static_cast<double>(points_number);
			converged_ = delta_p_norm == delta_p_norm;
			return;
		}

		delta_p.normalize();

		std::chrono::time_point<std::chrono::system_clock> computeStepLengthMT_start, computeStepLengthMT_end;
		computeStepLengthMT_start = std::chrono::system_clock::now();
		delta_p_norm = computeStepLengthMT(p, delta_p, delta_p_norm, step_size_, transformation_epsilon_ / 2, score, score_gradient, hessian, trans_cloud_);
		computeStepLengthMT_end = std::chrono::system_clock::now();
		double computeStepLengthMT_time = std::chrono::duration_cast<std::chrono::microseconds>(computeStepLengthMT_end - computeStepLengthMT_start).count() / 1000.0;
		std::cout << "computeStepLengthMT Time: " << computeStepLengthMT_time << " ms." << std::endl;

		delta_p *= delta_p_norm;

		transformation_ = (Eigen::Translation<float, 3>(static_cast<float>(delta_p(0)), static_cast<float>(delta_p(1)), static_cast<float>(delta_p(2))) *
							Eigen::AngleAxis<float>(static_cast<float>(delta_p(3)), Eigen::Vector3f::UnitX()) *
							Eigen::AngleAxis<float>(static_cast<float>(delta_p(4)), Eigen::Vector3f::UnitY()) *
							Eigen::AngleAxis<float>(static_cast<float>(delta_p(5)), Eigen::Vector3f::UnitZ())).matrix();

		p = p + delta_p;

		//Not update visualizer

		if (nr_iterations_ > max_iterations_ || (nr_iterations_ && (std::fabs(delta_p_norm) < transformation_epsilon_))) {
			converged_ = true;
		}

		nr_iterations_++;
	}

	if (source_cloud_->points.size() > 0) {
		trans_probability_ = score / static_cast<double>(source_cloud_->points.size());
	}
}


double NdtLib::computeStepLengthMT(const Eigen::Matrix<double, 6, 1> &x, Eigen::Matrix<double, 6, 1> &step_dir,
									double step_init, double step_max, double step_min, double &score,
									Eigen::Matrix<double, 6, 1> &score_gradient, Eigen::Matrix<double, 6, 6> &hessian,
									pcl::PointCloud<PointSourceType> &trans_cloud)
{
	double phi_0 = -score;
	double d_phi_0 = -(score_gradient.dot(step_dir));

	Eigen::Matrix<double, 6, 1> x_t;

	if (d_phi_0 >= 0) {
		if (d_phi_0 == 0) {
			return 0;
		} else {
			d_phi_0 *= -1;
			step_dir *= -1;
		}
	}

	int max_step_iterations = 10;
	int step_iterations = 0;

	double mu = 1.e-4;
	double nu = 0.9;
	double a_l = 0, a_u = 0;

	double f_l = auxilaryFunction_PsiMT(a_l, phi_0, phi_0, d_phi_0, mu);
	double g_l = auxilaryFunction_dPsiMT(d_phi_0, d_phi_0, mu);

	double f_u = auxilaryFunction_PsiMT(a_u, phi_0, phi_0, d_phi_0, mu);
	double g_u = auxilaryFunction_dPsiMT(d_phi_0, d_phi_0, mu);

	bool interval_converged = (step_max - step_min) > 0, open_interval = true;

	double a_t = step_init;
	a_t = std::min(a_t, step_max);
	a_t = std::max(a_t, step_min);

	x_t = x + step_dir * a_t;

	final_transformation_ = (Eigen::Translation<float, 3>(static_cast<float>(x_t(0)), static_cast<float>(x_t(1)), static_cast<float>(x_t(2))) *
								Eigen::AngleAxis<float>(static_cast<float>(x_t(3)), Eigen::Vector3f::UnitX()) *
								Eigen::AngleAxis<float>(static_cast<float>(x_t(4)), Eigen::Vector3f::UnitY()) *
								Eigen::AngleAxis<float>(static_cast<float>(x_t(5)), Eigen::Vector3f::UnitZ())).matrix();

	pcl::transformPointCloud(*source_cloud_, trans_cloud, final_transformation_);

	score = computeDerivatives(score_gradient, hessian, trans_cloud, x_t, true);

	double phi_t = -score;
	double d_phi_t = -(score_gradient.dot(step_dir));
	double psi_t = auxilaryFunction_PsiMT(a_t, phi_t, phi_0, d_phi_0, mu);
	double d_psi_t = auxilaryFunction_dPsiMT(d_phi_t, d_phi_0, mu);

	while (!interval_converged && step_iterations < max_step_iterations && !(psi_t <= 0 && d_phi_t <= -nu * d_phi_0)) {
		if (open_interval) {
			a_t = trialValueSelectionMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, psi_t, d_psi_t);
		} else {
			a_t = trialValueSelectionMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, phi_t, d_phi_t);
		}

		a_t = (a_t < step_max) ? a_t : step_max;
		a_t = (a_t > step_min) ? a_t : step_min;

		x_t = x + step_dir * a_t;

		final_transformation_ = (Eigen::Translation<float, 3>(static_cast<float>(x_t(0)), static_cast<float>(x_t(1)), static_cast<float>(x_t(2))) *
								 Eigen::AngleAxis<float>(static_cast<float>(x_t(3)), Eigen::Vector3f::UnitX()) *
								 Eigen::AngleAxis<float>(static_cast<float>(x_t(4)), Eigen::Vector3f::UnitY()) *
								 Eigen::AngleAxis<float>(static_cast<float>(x_t(5)), Eigen::Vector3f::UnitZ())).matrix();

		pcl::transformPointCloud(*source_cloud_, trans_cloud, final_transformation_);

		score = computeDerivatives(score_gradient, hessian, trans_cloud, x_t, false);

		phi_t -= score;
		d_phi_t -= (score_gradient.dot(step_dir));
		psi_t = auxilaryFunction_PsiMT(a_t, phi_t, phi_0, d_phi_0, mu);
		d_psi_t = auxilaryFunction_dPsiMT(d_phi_t, d_phi_0, mu);

		if (open_interval && (psi_t <= 0 && d_psi_t >= 0)) {
			open_interval = false;

			f_l += phi_0 - mu * d_phi_0 * a_l;
			g_l += mu * d_phi_0;

			f_u += phi_0 - mu * d_phi_0 * a_u;
			g_u += mu * d_phi_0;
		}

		if (open_interval) {
			interval_converged = updateIntervalMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, psi_t, d_psi_t);
		} else {
			interval_converged = updateIntervalMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, phi_t, d_phi_t);
		}
		step_iterations++;
	}

	if (step_iterations) {
		computeHessian(hessian, trans_cloud);
	}

	real_iterations_ += step_iterations;

	return a_t;
}

double NdtLib::auxilaryFunction_PsiMT(double a, double f_a, double f_0, double g_0, double mu)
{
  return (f_a - f_0 - mu * g_0 * a);
}

double NdtLib::auxilaryFunction_dPsiMT(double g_a, double g_0, double mu)
{
  return (g_a - mu * g_0);
}

double NdtLib::trialValueSelectionMT (double a_l, double f_l, double g_l,
									double a_u, double f_u, double g_u,
									double a_t, double f_t, double g_t)
{
	// Case 1 in Trial Value Selection [More, Thuente 1994]
	if (f_t > f_l) {
		// Calculate the minimizer of the cubic that interpolates f_l, f_t, g_l and g_t
		// Equation 2.4.52 [Sun, Yuan 2006]
		double z = 3 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
		double w = std::sqrt (z * z - g_t * g_l);
		// Equation 2.4.56 [Sun, Yuan 2006]
		double a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2 * w);

		// Calculate the minimizer of the quadratic that interpolates f_l, f_t and g_l
		// Equation 2.4.2 [Sun, Yuan 2006]
		double a_q = a_l - 0.5 * (a_l - a_t) * g_l / (g_l - (f_l - f_t) / (a_l - a_t));

		if (std::fabs (a_c - a_l) < std::fabs (a_q - a_l)) {
		  return (a_c);
		} else {
		  return (0.5 * (a_q + a_c));
		}
	}
	// Case 2 in Trial Value Selection [More, Thuente 1994]
	else if (g_t * g_l < 0) {
		// Calculate the minimizer of the cubic that interpolates f_l, f_t, g_l and g_t
		// Equation 2.4.52 [Sun, Yuan 2006]
		double z = 3 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
		double w = std::sqrt (z * z - g_t * g_l);
		// Equation 2.4.56 [Sun, Yuan 2006]
		double a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2 * w);

		// Calculate the minimizer of the quadratic that interpolates f_l, g_l and g_t
		// Equation 2.4.5 [Sun, Yuan 2006]
		double a_s = a_l - (a_l - a_t) / (g_l - g_t) * g_l;

		if (std::fabs (a_c - a_t) >= std::fabs (a_s - a_t)) {
		  return (a_c);
		} else {
		  return (a_s);
		}
	}
	// Case 3 in Trial Value Selection [More, Thuente 1994]
	else if (std::fabs (g_t) <= std::fabs (g_l)) {
		// Calculate the minimizer of the cubic that interpolates f_l, f_t, g_l and g_t
		// Equation 2.4.52 [Sun, Yuan 2006]
		double z = 3 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
		double w = std::sqrt (z * z - g_t * g_l);
		double a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2 * w);

		// Calculate the minimizer of the quadratic that interpolates g_l and g_t
		// Equation 2.4.5 [Sun, Yuan 2006]
		double a_s = a_l - (a_l - a_t) / (g_l - g_t) * g_l;

		double a_t_next;

		if (std::fabs (a_c - a_t) < std::fabs (a_s - a_t)) {
		  a_t_next = a_c;
		} else {
		  a_t_next = a_s;
		}

		if (a_t > a_l) {
		  return (std::min (a_t + 0.66 * (a_u - a_t), a_t_next));
		} else {
		  return (std::max (a_t + 0.66 * (a_u - a_t), a_t_next));
		}
	}
	// Case 4 in Trial Value Selection [More, Thuente 1994]
	else {
		// Calculate the minimizer of the cubic that interpolates f_u, f_t, g_u and g_t
		// Equation 2.4.52 [Sun, Yuan 2006]
		double z = 3 * (f_t - f_u) / (a_t - a_u) - g_t - g_u;
		double w = std::sqrt (z * z - g_t * g_u);
		// Equation 2.4.56 [Sun, Yuan 2006]
		return (a_u + (a_t - a_u) * (w - g_u - z) / (g_t - g_u + 2 * w));
	}
}

double NdtLib::computeDerivatives(Eigen::Matrix<double, 6, 1> &score_gradient, 
								Eigen::Matrix<double, 6, 6> &hessian,
								pcl::PointCloud<PointType> &trans_cloud,
								Eigen::Matrix<double, 6, 1> pose, 
								bool compute_hessian)
{

	PointSourceType x_pt, x_trans_pt;
	Eigen::Vector3d x, x_trans;
	Eigen::Matrix3d c_inv;

	score_gradient.setZero ();
	hessian.setZero ();

	//Compute Angle Derivatives
	computeAngleDerivatives(pose);

	std::vector<int> neighbor_ids;
	Eigen::Matrix<double, 3, 6> point_gradient;
	Eigen::Matrix<double, 18, 6> point_hessian;
	double score = 0;

	point_gradient.setZero();
	point_gradient.block<3, 3>(0, 0).setIdentity();
	point_hessian.setZero();

	for (size_t idx = 0; idx < source_cloud_->points.size(); idx++) {
		neighbor_ids.clear();
		x_trans_pt = trans_cloud.points[idx];

		voxel_grid_.radiusSearch(x_trans_pt, resolution_, neighbor_ids);

		for (size_t i = 0; i < neighbor_ids.size(); i++) {
			int vid = neighbor_ids[i];

			x_pt = source_cloud_->points[idx];
			x = Eigen::Vector3d(x_pt.x, x_pt.y, x_pt.z);

			x_trans = Eigen::Vector3d(x_trans_pt.x, x_trans_pt.y, x_trans_pt.z);

			x_trans -= voxel_grid_.getCentroid(vid);
			c_inv = voxel_grid_.getInverseCovariance(vid);

			computePointDerivatives(x, point_gradient, point_hessian, compute_hessian);

			score += updateDerivatives(score_gradient, hessian, point_gradient, point_hessian, x_trans, c_inv, compute_hessian);
		}
	}

	return score;
}

void NdtLib::computeAngleDerivatives(Eigen::Matrix<double, 6, 1> pose, bool compute_hessian)
{
	double cx, cy, cz, sx, sy, sz;

	if (fabs(pose(3)) < 10e-5) {
		cx = 1.0;
		sx = 0.0;
	} else {
		cx = cos(pose(3));
		sx = sin(pose(3));
	}

	if (fabs(pose(4)) < 10e-5) {
		cy = 1.0;
		sy = 0.0;
	} else {
		cy = cos(pose(4));
		sy = sin(pose(4));
	}

	if (fabs(pose(5)) < 10e-5) {
		cz = 1.0;
		sz = 0.0;
	} else {
		cz = cos(pose(5));
		sz = sin(pose(5));
	}

	j_ang_a_(0) = -sx * sz + cx * sy * cz;
	j_ang_a_(1) = -sx * cz - cx * sy * sz;
	j_ang_a_(2) = -cx * cy;

	j_ang_b_(0) = cx * sz + sx * sy * cz;
	j_ang_b_(1) = cx * cz - sx * sy * sz;
	j_ang_b_(2) = -sx * cy;

	j_ang_c_(0) = -sy * cz;
	j_ang_c_(1) = sy * sz;
	j_ang_c_(2) = cy;

	j_ang_d_(0) = sx * cy * cz;
	j_ang_d_(1) = -sx * cy * sz;
	j_ang_d_(2) = sx * sy;

	j_ang_e_(0) = -cx * cy * cz;
	j_ang_e_(1) = cx * cy * sz;
	j_ang_e_(2) = -cx * sy;

	j_ang_f_(0) = -cy * sz;
	j_ang_f_(1) = -cy * cz;
	j_ang_f_(2) = 0;

	j_ang_g_(0) = cx * cz - sx * sy * sz;
	j_ang_g_(1) = -cx * sz - sx * sy * cz;
	j_ang_g_(2) = 0;

	j_ang_h_(0) = sx * cz + cx * sy * sz;
	j_ang_h_(1) = cx * sy * cz - sx * sz;
	j_ang_h_(2) = 0;

	if (compute_hessian) {
		h_ang_a2_(0) = -cx * sz - sx * sy * cz;
		h_ang_a2_(1) = -cx * cz + sx * sy * sz;
		h_ang_a2_(2) = sx * cy;

		h_ang_a3_(0) = -sx * sz + cx * sy * cz;
		h_ang_a3_(1) = -cx * sy * sz - sx * cz;
		h_ang_a3_(2) = -cx * cy;

		h_ang_b2_(0) = cx * cy * cz;
		h_ang_b2_(1) = -cx * cy * sz;
		h_ang_b2_(2) = cx * sy;

		h_ang_b3_(0) = sx * cy * cz;
		h_ang_b3_(1) = -sx * cy * sz;
		h_ang_b3_(2) = sx * sy;

		h_ang_c2_(0) = -sx * cz - cx * sy * sz;
		h_ang_c2_(1) = sx * sz - cx * sy * cz;
		h_ang_c2_(2) = 0;

		h_ang_c3_(0) = cx * cz - sx * sy * sz;
		h_ang_c3_(1) = -sx * sy * cz - cx * sz;
		h_ang_c3_(2) = 0;

		h_ang_d1_(0) = -cy * cz;
		h_ang_d1_(1) = cy * sz;
		h_ang_d1_(2) = sy;

		h_ang_d2_(0) = -sx * sy * cz;
		h_ang_d2_(1) = sx * sy * sz;
		h_ang_d2_(2) = sx * cy;

		h_ang_d3_(0) = cx * sy * cz;
		h_ang_d3_(1) = -cx * sy * sz;
		h_ang_d3_(2) = -cx * cy;

		h_ang_e1_(0) = sy * sz;
		h_ang_e1_(1) = sy * cz;
		h_ang_e1_(2) = 0;

		h_ang_e2_(0) = -sx * cy * sz;
		h_ang_e2_(1) = -sx * cy * cz;
		h_ang_e2_(2) = 0;

		h_ang_e3_(0) = cx * cy * sz;
		h_ang_e3_(1) = cx * cy * cz;
		h_ang_e3_(2) = 0;

		h_ang_f1_(0) = -cy * cz;
		h_ang_f1_(1) = cy * sz;
		h_ang_f1_(2) = 0;

		h_ang_f2_(0) = -cx * sz - sx * sy * cz;
		h_ang_f2_(1) = -cx * cz + sx * sy * sz;
		h_ang_f2_(2) = 0;

		h_ang_f3_(0) = -sx * sz + cx * sy * cz;
		h_ang_f3_(1) = -cx * sy * sz - sx * cz;
		h_ang_f3_(2) = 0;
	}

}


//Copied from ndt.hpp
double NdtLib::updateIntervalMT (double &a_l, double &f_l, double &g_l,
								double &a_u, double &f_u, double &g_u,
								double a_t, double f_t, double g_t)
{
  // Case U1 in Update Algorithm and Case a in Modified Update Algorithm [More, Thuente 1994]
	if (f_t > f_l) {
		a_u = a_t;
		f_u = f_t;
		g_u = g_t;
		return (false);
	}
	// Case U2 in Update Algorithm and Case b in Modified Update Algorithm [More, Thuente 1994]
	else if (g_t * (a_l - a_t) > 0) {
		a_l = a_t;
		f_l = f_t;
		g_l = g_t;
		return (false);
	}
	// Case U3 in Update Algorithm and Case c in Modified Update Algorithm [More, Thuente 1994]
	else if (g_t * (a_l - a_t) < 0) {
		a_u = a_l;
		f_u = f_l;
		g_u = g_l;

		a_l = a_t;
		f_l = f_t;
		g_l = g_t;
		return (false);
	}
	// Interval Converged
	else {
		return (true);
	}
}


void NdtLib::computePointDerivatives(Eigen::Vector3d &x, Eigen::Matrix<double, 3, 6> &point_gradient, 
									Eigen::Matrix<double, 18, 6> &point_hessian, bool compute_hessian)
{
	point_gradient(1, 3) = x.dot(j_ang_a_);
	point_gradient(2, 3) = x.dot(j_ang_b_);
	point_gradient(0, 4) = x.dot(j_ang_c_);
	point_gradient(1, 4) = x.dot(j_ang_d_);
	point_gradient(2, 4) = x.dot(j_ang_e_);
	point_gradient(0, 5) = x.dot(j_ang_f_);
	point_gradient(1, 5) = x.dot(j_ang_g_);
	point_gradient(2, 5) = x.dot(j_ang_h_);

	if (compute_hessian) {
		Eigen::Vector3d a, b, c, d, e, f;

		a << 0, x.dot(h_ang_a2_), x.dot(h_ang_a3_);
		b << 0, x.dot(h_ang_b2_), x.dot(h_ang_b3_);
		c << 0, x.dot(h_ang_c2_), x.dot(h_ang_c3_);
		d << x.dot(h_ang_d1_), x.dot(h_ang_d2_), x.dot(h_ang_d3_);
		e << x.dot(h_ang_e1_), x.dot(h_ang_e2_), x.dot(h_ang_e3_);
		f << x.dot(h_ang_f1_), x.dot(h_ang_f2_), x.dot(h_ang_f3_);

		point_hessian.block<3, 1>(9, 3) = a;
		point_hessian.block<3, 1>(12, 3) = b;
		point_hessian.block<3, 1>(15, 3) = c;
		point_hessian.block<3, 1>(9, 4) = b;
		point_hessian.block<3, 1>(12, 4) = d;
		point_hessian.block<3, 1>(15, 4) = e;
		point_hessian.block<3, 1>(9, 5) = c;
		point_hessian.block<3, 1>(12, 5) = e;
		point_hessian.block<3, 1>(15, 5) = f;
	}
}

double NdtLib::updateDerivatives(Eigen::Matrix<double, 6, 1> &score_gradient, Eigen::Matrix<double, 6, 6> &hessian,
								Eigen::Matrix<double, 3, 6> point_gradient, Eigen::Matrix<double, 18, 6> point_hessian,
								Eigen::Vector3d &x_trans, Eigen::Matrix3d &c_inv, bool compute_hessian)
{
	Eigen::Vector3d cov_dxd_pi;
	double e_x_cov_x = exp(-gauss_d2_ * x_trans.dot(c_inv * x_trans) / 2);
	double score_inc = -gauss_d1_ * e_x_cov_x;

	e_x_cov_x = gauss_d2_ * e_x_cov_x;

	if (e_x_cov_x > 1 || e_x_cov_x < 0 || e_x_cov_x != e_x_cov_x) {
		return 0.0;
	}

	e_x_cov_x *= gauss_d1_;

	for (int i = 0; i < 6; i++) {
		cov_dxd_pi = c_inv * point_gradient.col(i);

		score_gradient(i) += x_trans.dot(cov_dxd_pi) * e_x_cov_x;

		if (compute_hessian) {
			for (int j = 0; j < hessian.cols(); j++) {
				hessian(i, j) += e_x_cov_x * (-gauss_d2_ * x_trans.dot(cov_dxd_pi) * x_trans.dot(c_inv * point_gradient.col(j)) +
									x_trans.dot(c_inv * point_hessian.block<3, 1>(3 * i, j)) +
									point_gradient.col(j).dot(cov_dxd_pi));
			}
		}
	}

	return score_inc;
}

void NdtLib::computeHessian(Eigen::Matrix<double, 6, 6> &hessian, 
							pcl::PointCloud<PointSourceType> &trans_cloud)
{
	PointSourceType x_pt, x_trans_pt;
	Eigen::Vector3d x, x_trans;
	Eigen::Matrix3d c_inv;

	hessian.setZero();

	Eigen::Matrix<double, 3, 6> point_gradient;
	Eigen::Matrix<double, 18, 6> point_hessian;


	for (size_t idx = 0; idx < source_cloud_->points.size(); idx++) {
		x_trans_pt = trans_cloud.points[idx];

		std::vector<int> neighbor_ids;

		voxel_grid_.radiusSearch(x_trans_pt, resolution_, neighbor_ids);

		for (size_t i = 0; i < neighbor_ids.size(); i++) {
			int vid = neighbor_ids[i];

			x_pt = source_cloud_->points[idx];
			x = Eigen::Vector3d(x_pt.x, x_pt.y, x_pt.z);
			x_trans = Eigen::Vector3d(x_trans_pt.x, x_trans_pt.y, x_trans_pt.z);
			x_trans -= voxel_grid_.getCentroid(vid);
			c_inv = voxel_grid_.getInverseCovariance(vid);

			computePointDerivatives(x, point_gradient, point_hessian);

			updateHessian(hessian, point_gradient, point_hessian, x_trans, c_inv);
		}
	}


}


void NdtLib::updateHessian(Eigen::Matrix<double, 6, 6> &hessian,
						Eigen::Matrix<double, 3, 6> point_gradient, 
						Eigen::Matrix<double, 18, 6> point_hessian,
						Eigen::Vector3d &x_trans, Eigen::Matrix3d &c_inv)
{
	Eigen::Vector3d cov_dxd_pi;
	double e_x_cov_x = gauss_d2_ * exp(-gauss_d2_ * x_trans.dot(c_inv * x_trans) / 2);

	if (e_x_cov_x > 1 || e_x_cov_x < 0 || e_x_cov_x != e_x_cov_x) {
		return;
	}

	e_x_cov_x *= gauss_d1_;

	for (int i = 0; i < 6; i++) {
		cov_dxd_pi = c_inv * point_gradient.col(i);

		for (int j = 0; j < hessian.cols(); j++) {
			hessian(i, j) += e_x_cov_x * (-gauss_d2_ * x_trans.dot(cov_dxd_pi) * x_trans.dot(c_inv * point_gradient.col(j)) +
								x_trans.dot(c_inv * point_hessian.block<3, 1>(3 * i, j)) +
								point_gradient.col(j).dot(cov_dxd_pi));
		}
	}
}



}  // namespace ndt_matching