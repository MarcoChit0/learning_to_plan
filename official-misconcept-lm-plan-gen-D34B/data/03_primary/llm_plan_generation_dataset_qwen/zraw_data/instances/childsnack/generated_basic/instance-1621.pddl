; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 633524

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child79 - child
    bread45 - bread-portion
    content358 - content-portion
    tray254 tray273 tray53 tray229 - tray
    table150 table414 table45 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray254 kitchen)
     (at tray273 kitchen)
     (at tray53 kitchen)
     (at tray229 kitchen)
     (at_kitchen_bread bread45)
     (at_kitchen_content content358)
     (not_allergic_gluten child79)
     (waiting child79 table414)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child79)
    )
  )
)
