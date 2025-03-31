(define (problem prob)
 (:domain barman)
 (:objects 
      shaker278 - shaker
      left right - hand
      shot373 shot96 shot333 - shot
      ingredient293 ingredient322 ingredient82 ingredient316 - ingredient
      cocktail1 - cocktail
      dispenser122 dispenser329 dispenser72 dispenser428 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker278)
  (ontable shot373)
  (ontable shot96)
  (ontable shot333)
  (dispenses dispenser122 ingredient293)
  (dispenses dispenser329 ingredient322)
  (dispenses dispenser72 ingredient82)
  (dispenses dispenser428 ingredient316)
  (clean shaker278)
  (clean shot373)
  (clean shot96)
  (clean shot333)
  (empty shaker278)
  (empty shot373)
  (empty shot96)
  (empty shot333)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker278 l0)
  (shaker-level shaker278 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient293)
  (cocktail-part2 cocktail1 ingredient82)
)
 (:goal
  (and
      (contains shot373 cocktail1)
      (contains shot96 cocktail1)
)))
