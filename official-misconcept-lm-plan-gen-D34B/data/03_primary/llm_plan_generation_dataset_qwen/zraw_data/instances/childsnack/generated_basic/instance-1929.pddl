; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 503949

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child303 - child
    bread270 - bread-portion
    content75 - content-portion
    tray355 tray173 tray343 - tray
    table60 table133 table247 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray355 kitchen)
     (at tray173 kitchen)
     (at tray343 kitchen)
     (at_kitchen_bread bread270)
     (at_kitchen_content content75)
     (not_allergic_gluten child303)
     (waiting child303 table133)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child303)
    )
  )
)
