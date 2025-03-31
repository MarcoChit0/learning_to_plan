(define (problem prob)
 (:domain barman)
 (:objects 
      shaker136 - shaker
      left right - hand
      shot437 shot124 shot278 - shot
      ingredient212 ingredient420 ingredient297 ingredient84 - ingredient
      cocktail305 - cocktail
      dispenser315 dispenser431 dispenser413 dispenser207 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker136)
  (ontable shot437)
  (ontable shot124)
  (ontable shot278)
  (dispenses dispenser315 ingredient212)
  (dispenses dispenser431 ingredient420)
  (dispenses dispenser413 ingredient297)
  (dispenses dispenser207 ingredient84)
  (clean shaker136)
  (clean shot437)
  (clean shot124)
  (clean shot278)
  (empty shaker136)
  (empty shot437)
  (empty shot124)
  (empty shot278)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker136 l0)
  (shaker-level shaker136 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail305 ingredient84)
  (cocktail-part2 cocktail305 ingredient297)
)
 (:goal
  (and
      (contains shot437 cocktail305)
      (contains shot124 cocktail305)
)))
