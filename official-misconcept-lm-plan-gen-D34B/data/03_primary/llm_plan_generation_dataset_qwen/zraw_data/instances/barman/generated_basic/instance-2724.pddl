(define (problem prob)
 (:domain barman)
 (:objects 
      shaker168 - shaker
      left right - hand
      shot144 shot402 shot453 - shot
      ingredient161 ingredient173 ingredient48 - ingredient
      cocktail263 - cocktail
      dispenser73 dispenser61 dispenser97 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker168)
  (ontable shot144)
  (ontable shot402)
  (ontable shot453)
  (dispenses dispenser73 ingredient161)
  (dispenses dispenser61 ingredient173)
  (dispenses dispenser97 ingredient48)
  (clean shaker168)
  (clean shot144)
  (clean shot402)
  (clean shot453)
  (empty shaker168)
  (empty shot144)
  (empty shot402)
  (empty shot453)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker168 l0)
  (shaker-level shaker168 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail263 ingredient173)
  (cocktail-part2 cocktail263 ingredient161)
)
 (:goal
  (and
      (contains shot144 cocktail263)
      (contains shot402 cocktail263)
)))
