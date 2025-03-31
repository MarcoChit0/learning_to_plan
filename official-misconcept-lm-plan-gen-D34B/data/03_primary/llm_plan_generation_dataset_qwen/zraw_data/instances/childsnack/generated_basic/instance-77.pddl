; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 634470

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child472 child265 - child
    bread245 bread84 - bread-portion
    content210 content370 - content-portion
    tray160 tray339 - tray
    table103 table308 table11 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray160 kitchen)
     (at tray339 kitchen)
     (at_kitchen_bread bread245)
     (at_kitchen_bread bread84)
     (at_kitchen_content content210)
     (at_kitchen_content content370)
     (not_allergic_gluten child472)
     (not_allergic_gluten child265)
     (waiting child472 table103)
     (waiting child265 table11)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child472)
     (served child265)
    )
  )
)
