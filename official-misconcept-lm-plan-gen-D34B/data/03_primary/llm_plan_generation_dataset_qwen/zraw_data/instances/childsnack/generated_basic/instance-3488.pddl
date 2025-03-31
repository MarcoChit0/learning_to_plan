; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 138219

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child152 child46 - child
    bread360 bread282 - bread-portion
    content210 content286 - content-portion
    tray330 tray99 - tray
    table380 table294 table161 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray330 kitchen)
     (at tray99 kitchen)
     (at_kitchen_bread bread360)
     (at_kitchen_bread bread282)
     (at_kitchen_content content210)
     (at_kitchen_content content286)
     (no_gluten_bread bread282)
     (no_gluten_content content210)
     (allergic_gluten child152)
     (not_allergic_gluten child46)
     (waiting child152 table294)
     (waiting child46 table294)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child152)
     (served child46)
    )
  )
)
