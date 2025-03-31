(define (problem prob)
 (:domain barman)
 (:objects 
      shaker321 - shaker
      left right - hand
      shot28 - shot
      ingredient292 ingredient443 ingredient26 ingredient164 - ingredient
      cocktail441 - cocktail
      dispenser339 dispenser347 dispenser153 dispenser263 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker321)
  (ontable shot28)
  (dispenses dispenser339 ingredient292)
  (dispenses dispenser347 ingredient443)
  (dispenses dispenser153 ingredient26)
  (dispenses dispenser263 ingredient164)
  (clean shaker321)
  (clean shot28)
  (empty shaker321)
  (empty shot28)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker321 l0)
  (shaker-level shaker321 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail441 ingredient26)
  (cocktail-part2 cocktail441 ingredient443)
)
 (:goal
  (and
      (contains shot28 cocktail441)
)))
