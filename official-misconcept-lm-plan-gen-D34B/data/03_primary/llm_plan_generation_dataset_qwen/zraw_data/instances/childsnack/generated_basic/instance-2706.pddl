; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 857299

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child173 child355 child167 - child
    bread125 bread245 bread356 - bread-portion
    content187 content284 content251 - content-portion
    tray65 tray244 tray53 - tray
    table56 table387 table448 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray65 kitchen)
     (at tray244 kitchen)
     (at tray53 kitchen)
     (at_kitchen_bread bread125)
     (at_kitchen_bread bread245)
     (at_kitchen_bread bread356)
     (at_kitchen_content content187)
     (at_kitchen_content content284)
     (at_kitchen_content content251)
     (not_allergic_gluten child167)
     (not_allergic_gluten child173)
     (not_allergic_gluten child355)
     (waiting child173 table387)
     (waiting child355 table448)
     (waiting child167 table56)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child173)
     (served child355)
     (served child167)
    )
  )
)
