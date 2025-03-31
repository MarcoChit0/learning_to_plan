; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 498586

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child69 child214 - child
    bread80 bread434 - bread-portion
    content100 content397 - content-portion
    tray477 tray490 tray351 - tray
    table103 table308 table218 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray477 kitchen)
     (at tray490 kitchen)
     (at tray351 kitchen)
     (at_kitchen_bread bread80)
     (at_kitchen_bread bread434)
     (at_kitchen_content content100)
     (at_kitchen_content content397)
     (not_allergic_gluten child69)
     (not_allergic_gluten child214)
     (waiting child69 table218)
     (waiting child214 table103)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child69)
     (served child214)
    )
  )
)
