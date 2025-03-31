(define (problem prob)
 (:domain barman)
 (:objects 
      shaker24 - shaker
      left right - hand
      shot278 - shot
      ingredient355 ingredient57 - ingredient
      cocktail422 - cocktail
      dispenser417 dispenser446 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker24)
  (ontable shot278)
  (dispenses dispenser417 ingredient355)
  (dispenses dispenser446 ingredient57)
  (clean shaker24)
  (clean shot278)
  (empty shaker24)
  (empty shot278)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker24 l0)
  (shaker-level shaker24 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail422 ingredient57)
  (cocktail-part2 cocktail422 ingredient355)
)
 (:goal
  (and
      (contains shot278 cocktail422)
)))
