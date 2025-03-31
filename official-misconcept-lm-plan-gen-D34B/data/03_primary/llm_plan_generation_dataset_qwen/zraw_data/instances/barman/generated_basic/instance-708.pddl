(define (problem prob)
 (:domain barman)
 (:objects 
      shaker278 - shaker
      left right - hand
      shot169 shot126 shot292 - shot
      ingredient464 ingredient342 ingredient484 ingredient86 - ingredient
      cocktail1 - cocktail
      dispenser211 dispenser77 dispenser113 dispenser126 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker278)
  (ontable shot169)
  (ontable shot126)
  (ontable shot292)
  (dispenses dispenser211 ingredient464)
  (dispenses dispenser77 ingredient342)
  (dispenses dispenser113 ingredient484)
  (dispenses dispenser126 ingredient86)
  (clean shaker278)
  (clean shot169)
  (clean shot126)
  (clean shot292)
  (empty shaker278)
  (empty shot169)
  (empty shot126)
  (empty shot292)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker278 l0)
  (shaker-level shaker278 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient464)
  (cocktail-part2 cocktail1 ingredient86)
)
 (:goal
  (and
      (contains shot169 cocktail1)
      (contains shot126 cocktail1)
)))
