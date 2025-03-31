(define (problem prob)
 (:domain barman)
 (:objects 
      shaker321 - shaker
      left right - hand
      shot476 shot260 shot4 - shot
      ingredient440 ingredient251 ingredient249 - ingredient
      cocktail148 - cocktail
      dispenser423 dispenser54 dispenser367 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker321)
  (ontable shot476)
  (ontable shot260)
  (ontable shot4)
  (dispenses dispenser423 ingredient440)
  (dispenses dispenser54 ingredient251)
  (dispenses dispenser367 ingredient249)
  (clean shaker321)
  (clean shot476)
  (clean shot260)
  (clean shot4)
  (empty shaker321)
  (empty shot476)
  (empty shot260)
  (empty shot4)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker321 l0)
  (shaker-level shaker321 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail148 ingredient249)
  (cocktail-part2 cocktail148 ingredient440)
)
 (:goal
  (and
      (contains shot476 cocktail148)
      (contains shot260 cocktail148)
)))
