(define (problem prob)
 (:domain barman)
 (:objects 
      shaker356 - shaker
      left right - hand
      shot89 shot73 - shot
      ingredient252 ingredient92 ingredient337 ingredient242 - ingredient
      cocktail190 - cocktail
      dispenser71 dispenser404 dispenser37 dispenser67 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker356)
  (ontable shot89)
  (ontable shot73)
  (dispenses dispenser71 ingredient252)
  (dispenses dispenser404 ingredient92)
  (dispenses dispenser37 ingredient337)
  (dispenses dispenser67 ingredient242)
  (clean shaker356)
  (clean shot89)
  (clean shot73)
  (empty shaker356)
  (empty shot89)
  (empty shot73)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker356 l0)
  (shaker-level shaker356 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail190 ingredient337)
  (cocktail-part2 cocktail190 ingredient242)
)
 (:goal
  (and
      (contains shot89 cocktail190)
)))
