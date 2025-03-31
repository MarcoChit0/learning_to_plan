; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 651412

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child231 - child
    bread88 - bread-portion
    content309 - content-portion
    tray384 tray159 tray150 tray452 - tray
    table34 table118 table212 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray384 kitchen)
     (at tray159 kitchen)
     (at tray150 kitchen)
     (at tray452 kitchen)
     (at_kitchen_bread bread88)
     (at_kitchen_content content309)
     (not_allergic_gluten child231)
     (waiting child231 table212)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child231)
    )
  )
)
