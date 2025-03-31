(define (problem prob)
 (:domain barman)
 (:objects 
      shaker170 - shaker
      left right - hand
      shot157 shot450 shot418 - shot
      ingredient107 ingredient332 ingredient379 - ingredient
      cocktail374 - cocktail
      dispenser363 dispenser185 dispenser16 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker170)
  (ontable shot157)
  (ontable shot450)
  (ontable shot418)
  (dispenses dispenser363 ingredient107)
  (dispenses dispenser185 ingredient332)
  (dispenses dispenser16 ingredient379)
  (clean shaker170)
  (clean shot157)
  (clean shot450)
  (clean shot418)
  (empty shaker170)
  (empty shot157)
  (empty shot450)
  (empty shot418)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker170 l0)
  (shaker-level shaker170 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail374 ingredient107)
  (cocktail-part2 cocktail374 ingredient379)
)
 (:goal
  (and
      (contains shot157 cocktail374)
      (contains shot450 ingredient332)
)))
