; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 448352

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child121 - child
    bread340 - bread-portion
    content16 - content-portion
    tray261 tray346 tray425 tray407 - tray
    table462 table304 table350 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray261 kitchen)
     (at tray346 kitchen)
     (at tray425 kitchen)
     (at tray407 kitchen)
     (at_kitchen_bread bread340)
     (at_kitchen_content content16)
     (not_allergic_gluten child121)
     (waiting child121 table462)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child121)
    )
  )
)
