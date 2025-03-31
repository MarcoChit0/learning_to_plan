(define (problem prob)
 (:domain barman)
 (:objects 
      shaker77 - shaker
      left right - hand
      shot275 shot337 shot351 - shot
      ingredient157 ingredient9 - ingredient
      cocktail278 - cocktail
      dispenser227 dispenser73 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker77)
  (ontable shot275)
  (ontable shot337)
  (ontable shot351)
  (dispenses dispenser227 ingredient157)
  (dispenses dispenser73 ingredient9)
  (clean shaker77)
  (clean shot275)
  (clean shot337)
  (clean shot351)
  (empty shaker77)
  (empty shot275)
  (empty shot337)
  (empty shot351)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker77 l0)
  (shaker-level shaker77 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail278 ingredient157)
  (cocktail-part2 cocktail278 ingredient9)
)
 (:goal
  (and
      (contains shot275 cocktail278)
      (contains shot337 cocktail278)
)))
