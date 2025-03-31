; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 200820

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child249 - child
    bread176 - bread-portion
    content255 - content-portion
    tray200 tray138 tray281 tray261 - tray
    table41 table94 table257 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray200 kitchen)
     (at tray138 kitchen)
     (at tray281 kitchen)
     (at tray261 kitchen)
     (at_kitchen_bread bread176)
     (at_kitchen_content content255)
     (not_allergic_gluten child249)
     (waiting child249 table41)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child249)
    )
  )
)
