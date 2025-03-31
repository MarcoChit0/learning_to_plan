; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 552866

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child124 - child
    bread138 - bread-portion
    content73 - content-portion
    tray360 tray341 - tray
    table470 table276 table219 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray360 kitchen)
     (at tray341 kitchen)
     (at_kitchen_bread bread138)
     (at_kitchen_content content73)
     (not_allergic_gluten child124)
     (waiting child124 table219)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child124)
    )
  )
)
