(define (problem prob)
 (:domain barman)
 (:objects 
      shaker127 - shaker
      left right - hand
      shot151 shot323 shot89 - shot
      ingredient78 ingredient102 - ingredient
      cocktail274 - cocktail
      dispenser91 dispenser367 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker127)
  (ontable shot151)
  (ontable shot323)
  (ontable shot89)
  (dispenses dispenser91 ingredient78)
  (dispenses dispenser367 ingredient102)
  (clean shaker127)
  (clean shot151)
  (clean shot323)
  (clean shot89)
  (empty shaker127)
  (empty shot151)
  (empty shot323)
  (empty shot89)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker127 l0)
  (shaker-level shaker127 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail274 ingredient78)
  (cocktail-part2 cocktail274 ingredient102)
)
 (:goal
  (and
      (contains shot151 cocktail274)
      (contains shot323 ingredient78)
)))
