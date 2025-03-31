; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 303245

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child107 - child
    bread380 - bread-portion
    content86 - content-portion
    tray353 tray60 tray206 tray332 - tray
    table158 table289 table89 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray353 kitchen)
     (at tray60 kitchen)
     (at tray206 kitchen)
     (at tray332 kitchen)
     (at_kitchen_bread bread380)
     (at_kitchen_content content86)
     (not_allergic_gluten child107)
     (waiting child107 table89)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child107)
    )
  )
)
