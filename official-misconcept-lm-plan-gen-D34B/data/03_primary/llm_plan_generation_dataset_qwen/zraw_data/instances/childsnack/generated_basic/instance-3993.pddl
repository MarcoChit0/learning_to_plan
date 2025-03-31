; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 697335

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child90 - child
    bread344 - bread-portion
    content108 - content-portion
    tray376 tray341 tray223 - tray
    table430 table157 table18 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray376 kitchen)
     (at tray341 kitchen)
     (at tray223 kitchen)
     (at_kitchen_bread bread344)
     (at_kitchen_content content108)
     (not_allergic_gluten child90)
     (waiting child90 table157)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child90)
    )
  )
)
