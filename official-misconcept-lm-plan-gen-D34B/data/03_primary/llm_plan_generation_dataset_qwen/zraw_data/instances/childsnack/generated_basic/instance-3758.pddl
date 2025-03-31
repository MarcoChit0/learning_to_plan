; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 326347

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child94 - child
    bread355 - bread-portion
    content62 - content-portion
    tray115 tray70 tray434 - tray
    table369 table168 table207 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray115 kitchen)
     (at tray70 kitchen)
     (at tray434 kitchen)
     (at_kitchen_bread bread355)
     (at_kitchen_content content62)
     (not_allergic_gluten child94)
     (waiting child94 table369)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child94)
    )
  )
)
