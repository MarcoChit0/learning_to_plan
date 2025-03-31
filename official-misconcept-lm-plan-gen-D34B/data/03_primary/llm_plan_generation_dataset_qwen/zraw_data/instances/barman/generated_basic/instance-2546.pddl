(define (problem prob)
 (:domain barman)
 (:objects 
      shaker235 - shaker
      left right - hand
      shot87 shot173 - shot
      ingredient22 ingredient143 - ingredient
      cocktail162 - cocktail
      dispenser365 dispenser346 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker235)
  (ontable shot87)
  (ontable shot173)
  (dispenses dispenser365 ingredient22)
  (dispenses dispenser346 ingredient143)
  (clean shaker235)
  (clean shot87)
  (clean shot173)
  (empty shaker235)
  (empty shot87)
  (empty shot173)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker235 l0)
  (shaker-level shaker235 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail162 ingredient22)
  (cocktail-part2 cocktail162 ingredient143)
)
 (:goal
  (and
      (contains shot87 cocktail162)
)))
