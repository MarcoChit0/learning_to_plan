; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 737925

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child383 child302 - child
    bread226 bread376 - bread-portion
    content202 content337 - content-portion
    tray384 tray41 - tray
    table293 table60 table370 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray384 kitchen)
     (at tray41 kitchen)
     (at_kitchen_bread bread226)
     (at_kitchen_bread bread376)
     (at_kitchen_content content202)
     (at_kitchen_content content337)
     (no_gluten_bread bread376)
     (no_gluten_content content337)
     (allergic_gluten child383)
     (not_allergic_gluten child302)
     (waiting child383 table293)
     (waiting child302 table370)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child383)
     (served child302)
    )
  )
)
