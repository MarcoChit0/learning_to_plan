(define (problem prob)
 (:domain barman)
 (:objects 
      shaker243 - shaker
      left right - hand
      shot137 shot413 shot391 - shot
      ingredient198 ingredient371 ingredient147 ingredient96 - ingredient
      cocktail218 - cocktail
      dispenser190 dispenser467 dispenser418 dispenser87 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker243)
  (ontable shot137)
  (ontable shot413)
  (ontable shot391)
  (dispenses dispenser190 ingredient198)
  (dispenses dispenser467 ingredient371)
  (dispenses dispenser418 ingredient147)
  (dispenses dispenser87 ingredient96)
  (clean shaker243)
  (clean shot137)
  (clean shot413)
  (clean shot391)
  (empty shaker243)
  (empty shot137)
  (empty shot413)
  (empty shot391)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker243 l0)
  (shaker-level shaker243 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail218 ingredient371)
  (cocktail-part2 cocktail218 ingredient198)
)
 (:goal
  (and
      (contains shot137 cocktail218)
      (contains shot413 ingredient96)
)))
