; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 868372

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child179 - child
    bread422 - bread-portion
    content10 - content-portion
    tray412 tray94 tray226 - tray
    table306 table345 table137 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray412 kitchen)
     (at tray94 kitchen)
     (at tray226 kitchen)
     (at_kitchen_bread bread422)
     (at_kitchen_content content10)
     (not_allergic_gluten child179)
     (waiting child179 table306)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child179)
    )
  )
)
