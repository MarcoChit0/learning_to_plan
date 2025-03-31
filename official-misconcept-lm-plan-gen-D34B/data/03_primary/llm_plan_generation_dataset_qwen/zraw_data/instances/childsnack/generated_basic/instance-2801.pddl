; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 654534

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child462 child59 - child
    bread73 bread85 - bread-portion
    content426 content403 - content-portion
    tray175 tray368 - tray
    table333 table275 table378 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray175 kitchen)
     (at tray368 kitchen)
     (at_kitchen_bread bread73)
     (at_kitchen_bread bread85)
     (at_kitchen_content content426)
     (at_kitchen_content content403)
     (not_allergic_gluten child59)
     (not_allergic_gluten child462)
     (waiting child462 table333)
     (waiting child59 table333)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child462)
     (served child59)
    )
  )
)
