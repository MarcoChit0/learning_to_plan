; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 321707

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child267 - child
    bread295 - bread-portion
    content293 - content-portion
    tray229 tray98 tray162 tray276 - tray
    table411 table485 table473 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray229 kitchen)
     (at tray98 kitchen)
     (at tray162 kitchen)
     (at tray276 kitchen)
     (at_kitchen_bread bread295)
     (at_kitchen_content content293)
     (not_allergic_gluten child267)
     (waiting child267 table473)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child267)
    )
  )
)
