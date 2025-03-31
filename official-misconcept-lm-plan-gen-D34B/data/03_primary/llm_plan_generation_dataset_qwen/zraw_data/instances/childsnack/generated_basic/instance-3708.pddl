; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 987618

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child69 child366 - child
    bread59 bread226 - bread-portion
    content315 content358 - content-portion
    tray490 tray419 - tray
    table364 table125 table236 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray490 kitchen)
     (at tray419 kitchen)
     (at_kitchen_bread bread59)
     (at_kitchen_bread bread226)
     (at_kitchen_content content315)
     (at_kitchen_content content358)
     (no_gluten_bread bread59)
     (no_gluten_content content315)
     (allergic_gluten child69)
     (not_allergic_gluten child366)
     (waiting child69 table125)
     (waiting child366 table125)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child69)
     (served child366)
    )
  )
)
