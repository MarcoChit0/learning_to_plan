; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 838035

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child475 child265 - child
    bread242 bread92 - bread-portion
    content89 content307 - content-portion
    tray363 - tray
    table182 table20 table339 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray363 kitchen)
     (at_kitchen_bread bread242)
     (at_kitchen_bread bread92)
     (at_kitchen_content content89)
     (at_kitchen_content content307)
     (no_gluten_bread bread92)
     (no_gluten_content content307)
     (allergic_gluten child265)
     (not_allergic_gluten child475)
     (waiting child475 table20)
     (waiting child265 table339)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child475)
     (served child265)
    )
  )
)
