; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 617442

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child149 - child
    bread498 - bread-portion
    content183 - content-portion
    tray188 tray332 tray410 - tray
    table226 table497 table435 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray188 kitchen)
     (at tray332 kitchen)
     (at tray410 kitchen)
     (at_kitchen_bread bread498)
     (at_kitchen_content content183)
     (not_allergic_gluten child149)
     (waiting child149 table226)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child149)
    )
  )
)
