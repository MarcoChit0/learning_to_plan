; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 211554

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child243 - child
    bread371 - bread-portion
    content434 - content-portion
    tray27 tray451 tray226 tray219 - tray
    table4 table464 table245 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray27 kitchen)
     (at tray451 kitchen)
     (at tray226 kitchen)
     (at tray219 kitchen)
     (at_kitchen_bread bread371)
     (at_kitchen_content content434)
     (not_allergic_gluten child243)
     (waiting child243 table245)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child243)
    )
  )
)
