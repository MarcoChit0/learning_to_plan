(define (problem prob)
 (:domain barman)
 (:objects 
      shaker174 - shaker
      left right - hand
      shot28 shot359 shot32 - shot
      ingredient278 ingredient168 ingredient223 ingredient401 - ingredient
      cocktail1 - cocktail
      dispenser297 dispenser359 dispenser96 dispenser433 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker174)
  (ontable shot28)
  (ontable shot359)
  (ontable shot32)
  (dispenses dispenser297 ingredient278)
  (dispenses dispenser359 ingredient168)
  (dispenses dispenser96 ingredient223)
  (dispenses dispenser433 ingredient401)
  (clean shaker174)
  (clean shot28)
  (clean shot359)
  (clean shot32)
  (empty shaker174)
  (empty shot28)
  (empty shot359)
  (empty shot32)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker174 l0)
  (shaker-level shaker174 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient223)
  (cocktail-part2 cocktail1 ingredient401)
)
 (:goal
  (and
      (contains shot28 cocktail1)
      (contains shot359 cocktail1)
)))
