(define (problem prob)
 (:domain barman)
 (:objects 
      shaker64 - shaker
      left right - hand
      shot341 shot367 shot16 - shot
      ingredient56 ingredient455 - ingredient
      cocktail303 - cocktail
      dispenser395 dispenser15 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker64)
  (ontable shot341)
  (ontable shot367)
  (ontable shot16)
  (dispenses dispenser395 ingredient56)
  (dispenses dispenser15 ingredient455)
  (clean shaker64)
  (clean shot341)
  (clean shot367)
  (clean shot16)
  (empty shaker64)
  (empty shot341)
  (empty shot367)
  (empty shot16)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker64 l0)
  (shaker-level shaker64 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail303 ingredient56)
  (cocktail-part2 cocktail303 ingredient455)
)
 (:goal
  (and
      (contains shot341 cocktail303)
      (contains shot367 cocktail303)
)))
