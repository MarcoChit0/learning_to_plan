(define (problem prob)
 (:domain barman)
 (:objects 
      shaker149 - shaker
      left right - hand
      shot338 shot211 shot91 - shot
      ingredient11 ingredient153 ingredient111 ingredient321 - ingredient
      cocktail1 - cocktail
      dispenser262 dispenser108 dispenser318 dispenser495 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker149)
  (ontable shot338)
  (ontable shot211)
  (ontable shot91)
  (dispenses dispenser262 ingredient11)
  (dispenses dispenser108 ingredient153)
  (dispenses dispenser318 ingredient111)
  (dispenses dispenser495 ingredient321)
  (clean shaker149)
  (clean shot338)
  (clean shot211)
  (clean shot91)
  (empty shaker149)
  (empty shot338)
  (empty shot211)
  (empty shot91)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker149 l0)
  (shaker-level shaker149 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient111)
  (cocktail-part2 cocktail1 ingredient11)
)
 (:goal
  (and
      (contains shot338 cocktail1)
      (contains shot211 cocktail1)
)))
