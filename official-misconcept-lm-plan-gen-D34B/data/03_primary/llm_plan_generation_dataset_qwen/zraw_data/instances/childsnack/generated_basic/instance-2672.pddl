; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 471248

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child432 - child
    bread435 - bread-portion
    content22 - content-portion
    tray313 tray422 tray428 tray70 - tray
    table275 table85 table369 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray313 kitchen)
     (at tray422 kitchen)
     (at tray428 kitchen)
     (at tray70 kitchen)
     (at_kitchen_bread bread435)
     (at_kitchen_content content22)
     (not_allergic_gluten child432)
     (waiting child432 table85)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child432)
    )
  )
)
