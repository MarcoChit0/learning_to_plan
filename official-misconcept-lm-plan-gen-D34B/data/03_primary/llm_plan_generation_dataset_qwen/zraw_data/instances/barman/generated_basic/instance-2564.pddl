(define (problem prob)
 (:domain barman)
 (:objects 
      shaker269 - shaker
      left right - hand
      shot375 shot367 shot379 - shot
      ingredient364 ingredient448 - ingredient
      cocktail227 - cocktail
      dispenser487 dispenser154 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker269)
  (ontable shot375)
  (ontable shot367)
  (ontable shot379)
  (dispenses dispenser487 ingredient364)
  (dispenses dispenser154 ingredient448)
  (clean shaker269)
  (clean shot375)
  (clean shot367)
  (clean shot379)
  (empty shaker269)
  (empty shot375)
  (empty shot367)
  (empty shot379)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker269 l0)
  (shaker-level shaker269 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail227 ingredient448)
  (cocktail-part2 cocktail227 ingredient364)
)
 (:goal
  (and
      (contains shot375 cocktail227)
      (contains shot367 ingredient448)
)))
