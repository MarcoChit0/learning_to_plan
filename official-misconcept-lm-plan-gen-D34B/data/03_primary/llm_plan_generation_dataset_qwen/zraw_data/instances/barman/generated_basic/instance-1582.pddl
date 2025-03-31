(define (problem prob)
 (:domain barman)
 (:objects 
      shaker339 - shaker
      left right - hand
      shot356 shot289 - shot
      ingredient143 ingredient146 - ingredient
      cocktail200 - cocktail
      dispenser458 dispenser222 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker339)
  (ontable shot356)
  (ontable shot289)
  (dispenses dispenser458 ingredient143)
  (dispenses dispenser222 ingredient146)
  (clean shaker339)
  (clean shot356)
  (clean shot289)
  (empty shaker339)
  (empty shot356)
  (empty shot289)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker339 l0)
  (shaker-level shaker339 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail200 ingredient143)
  (cocktail-part2 cocktail200 ingredient146)
)
 (:goal
  (and
      (contains shot356 cocktail200)
)))
