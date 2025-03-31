; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 671692

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child338 child466 - child
    bread273 bread29 - bread-portion
    content231 content313 - content-portion
    tray404 tray235 tray403 tray30 - tray
    table46 table51 table107 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray404 kitchen)
     (at tray235 kitchen)
     (at tray403 kitchen)
     (at tray30 kitchen)
     (at_kitchen_bread bread273)
     (at_kitchen_bread bread29)
     (at_kitchen_content content231)
     (at_kitchen_content content313)
     (not_allergic_gluten child338)
     (not_allergic_gluten child466)
     (waiting child338 table107)
     (waiting child466 table107)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child338)
     (served child466)
    )
  )
)
