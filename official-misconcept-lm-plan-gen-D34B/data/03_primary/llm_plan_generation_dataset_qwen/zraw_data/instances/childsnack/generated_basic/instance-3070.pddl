; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 996069

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child96 child468 - child
    bread115 bread255 - bread-portion
    content272 content239 - content-portion
    tray59 tray4 tray74 - tray
    table251 table365 table447 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray59 kitchen)
     (at tray4 kitchen)
     (at tray74 kitchen)
     (at_kitchen_bread bread115)
     (at_kitchen_bread bread255)
     (at_kitchen_content content272)
     (at_kitchen_content content239)
     (no_gluten_bread bread255)
     (no_gluten_content content239)
     (allergic_gluten child96)
     (not_allergic_gluten child468)
     (waiting child96 table365)
     (waiting child468 table447)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child96)
     (served child468)
    )
  )
)
