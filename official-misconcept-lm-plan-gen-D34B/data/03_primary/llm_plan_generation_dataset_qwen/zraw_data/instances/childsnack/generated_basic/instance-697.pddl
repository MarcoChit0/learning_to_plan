; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 641487

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child426 - child
    bread498 - bread-portion
    content241 - content-portion
    tray245 tray99 tray371 tray487 - tray
    table34 table9 table209 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray245 kitchen)
     (at tray99 kitchen)
     (at tray371 kitchen)
     (at tray487 kitchen)
     (at_kitchen_bread bread498)
     (at_kitchen_content content241)
     (not_allergic_gluten child426)
     (waiting child426 table209)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child426)
    )
  )
)
