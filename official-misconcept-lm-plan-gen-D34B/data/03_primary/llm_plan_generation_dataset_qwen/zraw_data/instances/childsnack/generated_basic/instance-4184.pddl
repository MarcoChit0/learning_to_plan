; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 625583

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child459 child60 - child
    bread379 bread167 - bread-portion
    content35 content214 - content-portion
    tray405 - tray
    table164 table98 table478 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray405 kitchen)
     (at_kitchen_bread bread379)
     (at_kitchen_bread bread167)
     (at_kitchen_content content35)
     (at_kitchen_content content214)
     (not_allergic_gluten child459)
     (not_allergic_gluten child60)
     (waiting child459 table164)
     (waiting child60 table98)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child459)
     (served child60)
    )
  )
)
