(define (problem prob)
 (:domain barman)
 (:objects 
      shaker236 - shaker
      left right - hand
      shot361 shot103 - shot
      ingredient73 ingredient265 ingredient188 - ingredient
      cocktail398 - cocktail
      dispenser312 dispenser315 dispenser8 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker236)
  (ontable shot361)
  (ontable shot103)
  (dispenses dispenser312 ingredient73)
  (dispenses dispenser315 ingredient265)
  (dispenses dispenser8 ingredient188)
  (clean shaker236)
  (clean shot361)
  (clean shot103)
  (empty shaker236)
  (empty shot361)
  (empty shot103)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker236 l0)
  (shaker-level shaker236 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail398 ingredient73)
  (cocktail-part2 cocktail398 ingredient188)
)
 (:goal
  (and
      (contains shot361 cocktail398)
)))
