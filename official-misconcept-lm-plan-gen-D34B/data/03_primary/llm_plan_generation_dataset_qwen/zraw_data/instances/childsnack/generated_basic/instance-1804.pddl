; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 822879

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child360 - child
    bread274 - bread-portion
    content338 - content-portion
    tray126 - tray
    table195 table6 table481 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray126 kitchen)
     (at_kitchen_bread bread274)
     (at_kitchen_content content338)
     (not_allergic_gluten child360)
     (waiting child360 table6)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child360)
    )
  )
)
