; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 484146

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child228 - child
    bread97 - bread-portion
    content279 - content-portion
    tray251 tray284 tray348 tray159 - tray
    table332 table414 table87 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray251 kitchen)
     (at tray284 kitchen)
     (at tray348 kitchen)
     (at tray159 kitchen)
     (at_kitchen_bread bread97)
     (at_kitchen_content content279)
     (not_allergic_gluten child228)
     (waiting child228 table332)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child228)
    )
  )
)
