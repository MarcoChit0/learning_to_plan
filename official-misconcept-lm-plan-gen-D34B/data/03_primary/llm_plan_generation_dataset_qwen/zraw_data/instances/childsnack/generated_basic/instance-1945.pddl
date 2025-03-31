; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 241226

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child156 - child
    bread200 - bread-portion
    content106 - content-portion
    tray428 tray26 tray266 - tray
    table358 table250 table340 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray428 kitchen)
     (at tray26 kitchen)
     (at tray266 kitchen)
     (at_kitchen_bread bread200)
     (at_kitchen_content content106)
     (not_allergic_gluten child156)
     (waiting child156 table250)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child156)
    )
  )
)
