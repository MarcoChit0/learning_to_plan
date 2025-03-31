; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 859430

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child15 - child
    bread380 - bread-portion
    content287 - content-portion
    tray262 tray216 tray414 tray257 - tray
    table266 table228 table485 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray262 kitchen)
     (at tray216 kitchen)
     (at tray414 kitchen)
     (at tray257 kitchen)
     (at_kitchen_bread bread380)
     (at_kitchen_content content287)
     (not_allergic_gluten child15)
     (waiting child15 table228)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child15)
    )
  )
)
