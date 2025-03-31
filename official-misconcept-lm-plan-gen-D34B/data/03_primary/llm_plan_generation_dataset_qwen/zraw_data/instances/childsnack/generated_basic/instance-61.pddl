; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 288699

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child148 child221 - child
    bread380 bread483 - bread-portion
    content152 content292 - content-portion
    tray412 tray385 - tray
    table173 table385 table462 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray412 kitchen)
     (at tray385 kitchen)
     (at_kitchen_bread bread380)
     (at_kitchen_bread bread483)
     (at_kitchen_content content152)
     (at_kitchen_content content292)
     (not_allergic_gluten child148)
     (not_allergic_gluten child221)
     (waiting child148 table385)
     (waiting child221 table462)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child148)
     (served child221)
    )
  )
)
