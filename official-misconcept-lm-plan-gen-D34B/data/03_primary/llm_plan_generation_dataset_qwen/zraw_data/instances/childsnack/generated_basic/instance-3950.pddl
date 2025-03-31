; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 901623

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child191 child184 - child
    bread441 bread386 - bread-portion
    content256 content161 - content-portion
    tray463 - tray
    table293 table267 table330 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray463 kitchen)
     (at_kitchen_bread bread441)
     (at_kitchen_bread bread386)
     (at_kitchen_content content256)
     (at_kitchen_content content161)
     (no_gluten_bread bread441)
     (no_gluten_content content161)
     (allergic_gluten child184)
     (not_allergic_gluten child191)
     (waiting child191 table330)
     (waiting child184 table330)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child191)
     (served child184)
    )
  )
)
