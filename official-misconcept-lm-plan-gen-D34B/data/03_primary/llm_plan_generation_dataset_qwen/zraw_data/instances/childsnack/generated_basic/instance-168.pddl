; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 150627

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child270 child244 - child
    bread114 bread146 - bread-portion
    content9 content319 - content-portion
    tray484 tray384 tray467 tray302 - tray
    table57 table137 table337 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray484 kitchen)
     (at tray384 kitchen)
     (at tray467 kitchen)
     (at tray302 kitchen)
     (at_kitchen_bread bread114)
     (at_kitchen_bread bread146)
     (at_kitchen_content content9)
     (at_kitchen_content content319)
     (not_allergic_gluten child270)
     (not_allergic_gluten child244)
     (waiting child270 table337)
     (waiting child244 table137)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child270)
     (served child244)
    )
  )
)
