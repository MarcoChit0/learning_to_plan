(define (problem prob)
 (:domain barman)
 (:objects 
      shaker459 - shaker
      left right - hand
      shot23 shot467 shot344 - shot
      ingredient198 ingredient464 ingredient56 ingredient466 - ingredient
      cocktail1 - cocktail
      dispenser96 dispenser49 dispenser390 dispenser412 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker459)
  (ontable shot23)
  (ontable shot467)
  (ontable shot344)
  (dispenses dispenser96 ingredient198)
  (dispenses dispenser49 ingredient464)
  (dispenses dispenser390 ingredient56)
  (dispenses dispenser412 ingredient466)
  (clean shaker459)
  (clean shot23)
  (clean shot467)
  (clean shot344)
  (empty shaker459)
  (empty shot23)
  (empty shot467)
  (empty shot344)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker459 l0)
  (shaker-level shaker459 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient466)
  (cocktail-part2 cocktail1 ingredient56)
)
 (:goal
  (and
      (contains shot23 cocktail1)
      (contains shot467 cocktail1)
)))
