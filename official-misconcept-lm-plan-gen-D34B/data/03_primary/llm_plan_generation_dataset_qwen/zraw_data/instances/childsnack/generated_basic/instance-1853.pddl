; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 683899

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child75 child35 - child
    bread460 bread183 - bread-portion
    content357 content480 - content-portion
    tray418 - tray
    table422 table39 table347 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray418 kitchen)
     (at_kitchen_bread bread460)
     (at_kitchen_bread bread183)
     (at_kitchen_content content357)
     (at_kitchen_content content480)
     (no_gluten_bread bread460)
     (no_gluten_content content480)
     (allergic_gluten child75)
     (not_allergic_gluten child35)
     (waiting child75 table39)
     (waiting child35 table347)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child75)
     (served child35)
    )
  )
)
